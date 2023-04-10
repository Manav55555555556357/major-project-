import { initializeApp } from "firebase/app";
import { getAuth, updateCurrentUser } from 'firebase/auth';
import { signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut, updateProfile } from "firebase/auth";
import { getFirestore, collection, doc, getDocs, getDoc, updateDoc, addDoc, query, where, setDoc, orderBy, writeBatch } from "firebase/firestore";
import { getStorage, ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { Donation } from "../interfaces";

const app = initializeApp({
    apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
    authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
    projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
    storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
    appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
    measurementId: process.env.NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID,
});

export const auth = getAuth(app);
export const firestore = getFirestore(app);
export const storage = getStorage(app);

export const signUp = async (email: string, password: string, userData) => {
    const user = await createUserWithEmailAndPassword(auth, email, password);
    await setDoc(doc(firestore, "users", auth.currentUser.uid), {
        uid: auth.currentUser.uid,
        ...userData
    })
    return user
};

export const logIn = (email: string, password: string) => {
    return signInWithEmailAndPassword(auth, email, password);
};

export const getUserDetails = async () => {
    return getDoc(doc(firestore, "users", auth.currentUser.uid))
}

export const getRestaurants = async () => {
    const q = query(collection(firestore, 'users'), where('userType', '==', 'Restaurant'));
    const querySnapshot = await getDocs(q)
    const restaurants = [];
    querySnapshot.forEach((doc) => {
        restaurants.push({ id: doc.id, ...doc.data() });
    });
    return restaurants;
};

export const logOut = async () => {
    await signOut(auth);
};

// CRUD operations for donations
export const createDonation = async (restaurantId: string, foodName: string, quantity: number, imageFile: File) => {
    const donationRef = doc(collection(firestore, 'donations'));
    const storageRef = ref(storage, `donation-images/${donationRef.id}`);
    await uploadBytes(storageRef, imageFile);
    const imageUrl = await getDownloadURL(storageRef);

    await setDoc(donationRef, {
        restaurantId,
        foodName,
        quantity,
        imageUrl,
        status: "available",
        timestamp: new Date()
    });

    return { donation: donationRef.id, imageUrl };
};

export const createDonationRequest = async (ngoId: string, donationIds: string[]) => {
    const batch = writeBatch(firestore);

    const requestRefs = [];

    for (const donationId of donationIds) {
        // Create the donation request document in the 'requests' collection
        const requestRef = doc(collection(firestore, 'requests'));
        batch.set(requestRef, {
            ngoId,
            donationId,
            status: "pending",
            timestamp: new Date()
        });
        requestRefs.push(requestRef);

        // Update the donation status in the 'donations' collection
        const donationRef = doc(firestore, 'donations', donationId);
        batch.update(donationRef, {
            status: "requested"
        });
    }

    await batch.commit();

    return requestRefs.map((ref) => ref.id);
};

export const getRestaurantDonations = async (restaurantId: string) => {
    const donationsRef = collection(firestore, "donations");
    const q = query(
        donationsRef,
        where("restaurantId", "==", restaurantId),
        where("status", "==", "available"),
        orderBy("timestamp", "desc")
    );

    const querySnapshot = await getDocs(q);
    const donations: Donation[] = [];
    querySnapshot.forEach((doc) => {
        donations.push({ id: doc.id, ...doc.data() } as Donation);
    });

    return donations;
};

export const updateDonation = async (donationId: string, updatedData) => {
    const donationRef = doc(firestore, 'donations', donationId);
    await updateDoc(donationRef, updatedData);
};

export const getRequestedDonations = async (restaurantId: string) => {
    const requestsSnapshot = await getDocs(query(collection(firestore, 'requests'), where('status', '==', 'pending')));
    const requests = [];

    for (const requestDoc of requestsSnapshot.docs) {
        const donationRef = doc(firestore, 'donations', requestDoc.data().donationId);
        const donationSnapshot = await getDoc(donationRef);

        if (donationSnapshot.exists() && donationSnapshot.data().restaurantId === restaurantId) {
            const ngoRef = doc(firestore, 'users', requestDoc.data().ngoId);
            const ngoSnapshot = await getDoc(ngoRef);
            // console.log(ngoSnapshot.data())

            requests.push({
                requestId: requestDoc.id,
                donationId: requestDoc.data().donationId,
                ngoId: requestDoc.data().ngoId,
                ngoName: ngoSnapshot.data().userData["NGO Name"],
                foodName: donationSnapshot.data().foodName,
                quantity: donationSnapshot.data().quantity,
            });
        }
    }

    return requests;
};

export const acceptDonationRequest = async (requestId: string, donationId: string) => {
    // Update the donation request status in the 'requests' collection
    const requestRef = doc(firestore, 'requests', requestId);
    await updateDoc(requestRef, {
        status: "accepted"
    });

    // Update the donation status in the 'donations' collection
    const donationRef = doc(firestore, 'donations', donationId);
    await updateDoc(donationRef, {
        status: "accepted"
    });
};

export const declineDonationRequest = async (requestId: string, donationId: string) => {
    // Update the donation request status in the 'requests' collection
    const requestRef = doc(firestore, 'requests', requestId);
    await updateDoc(requestRef, {
        status: "declined"
    });

    // Update the donation status in the 'donations' collection
    const donationRef = doc(firestore, 'donations', donationId);
    await updateDoc(donationRef, {
        status: "declined"
    });
};
